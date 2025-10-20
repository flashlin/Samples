using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class SettleActionEntityConfiguration : IEntityTypeConfiguration<SettleActionEntity>
    {
        public void Configure(EntityTypeBuilder<SettleActionEntity> builder)
        {
            builder.ToTable("SettleAction");

            builder.HasKey(x => x.ActionID);

            builder.Property(x => x.ActionID)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.StoredProcedure)
                .HasColumnType("nvarchar(200)")
                .IsRequired()
                .HasMaxLength(200)
            ;

            builder.Property(x => x.ParamNames)
                .HasColumnType("nvarchar(2000)")
                .IsRequired()
                .HasMaxLength(2000)
            ;

            builder.Property(x => x.ParamValues)
                .HasColumnType("nvarchar(2000)")
                .IsRequired()
                .HasMaxLength(2000)
            ;

            builder.Property(x => x.Creator)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.CreatedOn)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.TStamp)
                .HasColumnType("timestamp")
                .IsRequired()
            ;

        }
    }
}
