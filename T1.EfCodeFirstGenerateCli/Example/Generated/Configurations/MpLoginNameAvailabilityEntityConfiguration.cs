using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class MpLoginNameAvailabilityEntityConfiguration : IEntityTypeConfiguration<MpLoginNameAvailabilityEntity>
    {
        public void Configure(EntityTypeBuilder<MpLoginNameAvailabilityEntity> builder)
        {
            builder.ToTable("MpLoginNameAvailability");

            builder.HasKey(x => x.id);

            builder.Property(x => x.id)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.server)
                .HasColumnType("varchar(20)")
                .HasMaxLength(20)
            ;

            builder.Property(x => x.UsrNameOrSessionID)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.LoginName)
                .HasColumnType("nvarchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.GrantTime)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.Taken)
                .HasColumnType("bit")
                .IsRequired()
            ;

            builder.Property(x => x.ClientIP)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

        }
    }
}
