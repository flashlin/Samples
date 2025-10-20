using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class LoginBlackListEntityConfiguration : IEntityTypeConfiguration<LoginBlackListEntity>
    {
        public void Configure(EntityTypeBuilder<LoginBlackListEntity> builder)
        {
            builder.ToTable("LoginBlackList");

            builder.HasKey(x => x.LoginBlackListId);

            builder.Property(x => x.LoginBlackListId)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.CountryCode)
                .HasColumnType("char(2)")
                .HasMaxLength(2)
            ;

            builder.Property(x => x.CurrencyId)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.CreatedOn)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.CreatedBy)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

        }
    }
}
